"""Causal, timestamp-aware temporal evidence for a pretrained UIV-US head."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalTemporalObserver(nn.Module):
    """Add signed temporal detail and an adaptive local-trend observer.

    Input ``features`` has shape [B,T,D]. ``timestamps`` has shape [B,T]
    and uses any *consistent* time unit; omission means frame units (dt=1).
    ``valid`` is boolean [B,T]. Invalid slots advance time without a
    measurement update; their input feature is echoed and cannot affect later
    observations. ``quality`` is [B,T] in [0,1], where 1 means more reliable
    and omission is neutral (0.5). Modality is integer [B]: 0=B-mode,
    1=CEUS, or None for an unspecified modality. Modality noise adjustments
    start at zero and do not encode a claimed physical noise distinction.

    Haar scales count consecutive FRAME SLOTS, not seconds or Hz. At scale s,
    the coefficient is (sum(last s)-sum(previous s))/sqrt(2*s). A coefficient
    needs 2*s valid past/current slots, with no detected large cadence gap.
    This is a causal frame-scale detail transform, not an irregular-time FFT.
    All timestamp differences must be positive, including at invalid slots.
    Huge dt values saturate at 1e6 time units for numerical safety; no reliable
    interpolation through an unobserved long interval is implied.

    Returns ``features`` [B,T,D], filtered position ``state`` [B,T,S],
    pre-update ``innovation`` [B,T,S], position Kalman-style ``gain``
    [B,T,S], signed ``wavelet`` [B,T,len(scales),S], and ``spectral`` with
    the same shape (zero outside STFT mode). Innovation, gain and transform
    features are zero at invalid slots. The latent state is an estimate, not
    a physiological measurement or calibrated diagnostic uncertainty.

    The residual projection starts at zero, preserving pretrained features.
    The first slot ALWAYS has zero residual, so T=1 remains an exact no-op
    after training and agrees with the first prefix of a longer sequence.
    ``mode='fixed'`` uses neutral quality and fixed Q/R, and excludes Haar
    bands from fusion. ``mode='wavelet'`` fuses only the signed Haar bands;
    its state diagnostics are still returned. ``adaptive`` is the full path.
    ``mode='stft'`` instead fuses causal, phase-insensitive log powers from
    actual rFFTs: each complete 2*s frame window is demeaned, periodic-Hann
    windowed, and its first positive frequency bin is used. Its center is
    1/(2*s) cycles/frame, NOT Hz; no irregular-time resampling is performed.
    Spectral power is N*(real(FFT[1])**2+imag(FFT[1])**2)/sum(Hann**2),
    where FFT uses orthonormal normalization; the feature is log1p(power).
    It uses the same valid/gap guards and residual dimensions as Haar mode.
    """

    def __init__(
        self,
        feature_dim: int,
        state_dim: int = 16,
        scales: Sequence[int] = (1, 2, 4),
        mode: str = "adaptive",
    ) -> None:
        super().__init__()
        for name, value in (("feature_dim", feature_dim), ("state_dim", state_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        try:
            scales = tuple(scales)
        except TypeError as exc:
            raise ValueError("scales must be a nonempty sequence of positive integers") from exc
        if not scales or any(
            isinstance(s, bool) or not isinstance(s, int) or s <= 0 for s in scales
        ):
            raise ValueError("scales must be a nonempty sequence of positive integers")
        if len(set(scales)) != len(scales):
            raise ValueError("scales must not contain duplicates")
        if mode not in ("adaptive", "fixed", "wavelet", "stft"):
            raise ValueError("mode must be 'adaptive', 'fixed', 'wavelet', or 'stft'")

        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.scales = scales
        self.mode = mode
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.state_encoder = nn.Linear(feature_dim, state_dim)

        # Positive physical-time damping and continuous process-noise rates.
        self.raw_decay = nn.Parameter(torch.full((state_dim,), self._inv_softplus(0.45)))
        self.raw_acceleration_noise = nn.Parameter(
            torch.full((state_dim,), self._inv_softplus(0.02))
        )
        self.raw_position_noise = nn.Parameter(
            torch.full((state_dim,), self._inv_softplus(0.002))
        )
        self.raw_observation_noise = nn.Parameter(
            torch.full((state_dim,), self._inv_softplus(0.1))
        )
        self.modality_log_noise = nn.Embedding(3, 2)
        nn.init.zeros_(self.modality_log_noise.weight)

        # Learn bounded corrections to effective Q and R from signed detail
        # and innovations. No frequency is hard-coded as noise.
        self.noise_controller = nn.Linear((len(scales) + 2) * state_dim, 2 * state_dim)
        nn.init.zeros_(self.noise_controller.weight)
        nn.init.zeros_(self.noise_controller.bias)
        # Position, velocity, innovation, and all signed Haar scales.
        self.residual_projection = nn.Linear((len(scales) + 3) * state_dim, feature_dim)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    @staticmethod
    def _inv_softplus(value: float) -> float:
        return math.log(math.expm1(value))

    @staticmethod
    def _finite(name: str, value: Tensor) -> None:
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")

    @staticmethod
    def _stabilize_covariance(covariance: Tensor) -> Tensor:
        """Bound variance and roundoff correlation while preserving PSD."""
        p00 = covariance[..., 0, 0].clamp(1e-6, 1e6)
        p11 = covariance[..., 1, 1].clamp(1e-6, 1e6)
        bound = (p00 * p11).sqrt() * (1.0 - 1e-5)
        p01 = (0.5 * (covariance[..., 0, 1] + covariance[..., 1, 0]))
        p01 = torch.maximum(torch.minimum(p01, bound), -bound)
        return torch.stack(
            (torch.stack((p00, p01), -1), torch.stack((p01, p11), -1)), -2
        )

    def _transition_and_noise(
        self, dt: Tensor, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        """Exact damped-velocity transition and integrated diffusion Q.

        Per latent channel, dp=v*dt + position diffusion and
        dv=-decay*v*dt + acceleration diffusion. A small-argument expansion
        avoids cancellation in the integrated position variance.
        """
        decay = (F.softplus(self.raw_decay.to(dtype)) + 0.05).clamp(max=20.0)
        acc = (F.softplus(self.raw_acceleration_noise.to(dtype)) + 1e-6).clamp(max=10.0)
        pos = (F.softplus(self.raw_position_noise.to(dtype)) + 1e-6).clamp(max=10.0)
        d = dt[:, None].clamp(max=1e6)
        x = d * decay[None, :]
        b = torch.exp(-x)
        a = -torch.expm1(-x) / decay[None, :]
        one = torch.ones_like(a)
        zero = torch.zeros_like(a)
        transition = torch.stack(
            (torch.stack((one, a), -1), torch.stack((zero, b), -1)), -2
        )
        ivv = -torch.expm1(-2.0 * x) / (2.0 * decay[None, :])
        ipv = 0.5 * a.square()
        ipp_main = ((d - 2.0 * a + ivv) / decay[None, :].square()).clamp_min(0.0)
        # Cap the unused series branch too, preventing inf*0 in backward.
        small_d = torch.minimum(d.expand_as(x), 0.05 / decay[None, :])
        small_x = small_d * decay[None, :]
        ipp_series = small_d.pow(3) * (
            1.0 / 3.0 - small_x / 4.0 + 7.0 * small_x.square() / 60.0
            - small_x.pow(3) / 24.0
        )
        ipp = torch.where(x < 0.05, ipp_series, ipp_main)
        q00 = acc[None, :] * ipp + pos[None, :] * d
        q01 = acc[None, :] * ipv
        q11 = acc[None, :] * ivv
        noise = torch.stack(
            (torch.stack((q00, q01), -1), torch.stack((q01, q11), -1)), -2
        )
        return transition, noise

    def forward(
        self,
        features: Tensor,
        timestamps: Optional[Tensor] = None,
        valid: Optional[Tensor] = None,
        quality: Optional[Tensor] = None,
        modality: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if not isinstance(features, Tensor) or features.ndim != 3:
            raise ValueError("features must be a floating tensor of shape [B,T,D]")
        if not features.is_floating_point():
            raise ValueError("features must be floating point")
        batch, length, dimension = features.shape
        if batch < 1 or length < 1 or dimension != self.feature_dim:
            raise ValueError(f"features must have nonempty shape [B,T,{self.feature_dim}]")
        self._finite("features", features)
        device = features.device
        if self.state_encoder.weight.device != device:
            raise ValueError("features and the observer parameters must be on the same device")
        dtype = torch.float64 if features.dtype == torch.float64 else torch.float32

        if valid is None:
            valid = torch.ones((batch, length), device=device, dtype=torch.bool)
        elif not isinstance(valid, Tensor) or valid.shape != (batch, length) or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean tensor of shape [B,T]")
        else:
            valid = valid.to(device=device)

        if timestamps is None:
            times = torch.arange(length, device=device, dtype=torch.float64).expand(batch, -1)
        else:
            if (
                not isinstance(timestamps, Tensor) or timestamps.shape != (batch, length)
                or timestamps.dtype == torch.bool or timestamps.is_complex()
            ):
                raise ValueError("timestamps must be a real tensor of shape [B,T]")
            self._finite("timestamps", timestamps)
            # Subtract before reducing precision: large absolute timestamps can
            # still contain useful small relative intervals.
            times = timestamps.to(device=device, dtype=torch.float64)
        intervals = times[:, 1:] - times[:, :-1]
        if not bool(torch.isfinite(intervals).all()) or not bool((intervals > 0).all()):
            raise ValueError("timestamps must be finite and strictly increasing in every row")
        intervals = intervals.clamp(max=1e6).to(dtype=dtype)

        if quality is None:
            qualities = torch.full((batch, length), 0.5, device=device, dtype=dtype)
        else:
            if (
                not isinstance(quality, Tensor) or quality.shape != (batch, length)
                or not quality.is_floating_point()
            ):
                raise ValueError("quality must be a floating tensor of shape [B,T]")
            self._finite("quality", quality)
            if not bool(((quality >= 0) & (quality <= 1)).all()):
                raise ValueError("quality values must be in [0,1]")
            qualities = quality.to(device=device, dtype=dtype)

        if modality is None:
            modes = torch.full((batch,), 2, device=device, dtype=torch.long)
        else:
            integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            if (
                not isinstance(modality, Tensor) or modality.shape != (batch,)
                or modality.dtype not in integer_dtypes
            ):
                raise ValueError("modality must be integer [B], with 0=B-mode or 1=CEUS")
            if not bool(((modality == 0) | (modality == 1)).all()):
                raise ValueError("modality values must be 0 (B-mode) or 1 (CEUS)")
            modes = modality.to(device=device, dtype=torch.long)

        encoded = torch.tanh(self.state_encoder(
            self.feature_norm(features.to(dtype=self.state_encoder.weight.dtype))
        )).to(dtype=dtype)
        encoded = torch.where(valid[..., None], encoded, torch.zeros_like(encoded))
        mode_noise = self.modality_log_noise(modes).to(dtype=dtype).clamp(-2.0, 2.0)
        base_r = (F.softplus(self.raw_observation_noise.to(dtype)) + 1e-4).clamp(max=1e4)
        mean = torch.zeros((batch, self.state_dim, 2), device=device, dtype=dtype)
        identity = torch.eye(2, device=device, dtype=dtype).expand(batch, self.state_dim, 2, 2)
        covariance = identity.clone()
        previous_innovation = torch.zeros((batch, self.state_dim), device=device, dtype=dtype)
        cadence = torch.ones(batch, device=device, dtype=dtype)
        have_cadence = torch.zeros(batch, device=device, dtype=torch.bool)
        gap_edges = []
        states, velocities, innovations, gains, details, spectra = [], [], [], [], [], []
        spectral_windows = {
            scale: torch.hann_window(2 * scale, periodic=True, device=device, dtype=dtype)
            for scale in self.scales
        } if self.mode == "stft" else {}

        for t in range(length):
            observed = valid[:, t]
            dt = torch.zeros(batch, device=device, dtype=dtype) if t == 0 else intervals[:, t - 1]
            gap = have_cadence & (dt > 8.0 * cadence)
            gap_edges.append(gap)
            if t > 0:
                pair_valid = valid[:, t - 1] & observed
                # Robust causal cadence reference: isolated long gaps cannot
                # redefine the cadence and conceal another long gap.
                robust_dt = torch.maximum(torch.minimum(dt, 2.0 * cadence), 0.5 * cadence)
                updated = torch.where(have_cadence, 0.9 * cadence + 0.1 * robust_dt, dt)
                cadence = torch.where(pair_valid, updated, cadence)
                have_cadence = have_cadence | pair_valid

            coefficients, spectral_coefficients = [], []
            for scale in self.scales:
                start = t + 1 - 2 * scale
                spectral_coefficient = torch.zeros_like(previous_innovation)
                if start < 0:
                    coefficient = torch.zeros_like(previous_innovation)
                else:
                    window = encoded[:, start:t + 1]
                    coefficient = (
                        window[:, scale:].sum(1) - window[:, :scale].sum(1)
                    ) / math.sqrt(2.0 * scale)
                    complete = valid[:, start:t + 1].all(1)
                    crosses_gap = torch.stack(gap_edges[start + 1:t + 1], 1).any(1)
                    coefficient = torch.where(
                        (complete & ~crosses_gap)[:, None], coefficient,
                        torch.zeros_like(coefficient),
                    )
                    if self.mode == "stft":
                        # This transform is along TIME (dimension 1), never
                        # across a channel or image spatial dimension.
                        centered = window - window.mean(1, keepdim=True)
                        hann = spectral_windows[scale]
                        spectrum = torch.fft.rfft(
                            centered * hann[None, :, None], dim=1, norm="ortho"
                        )
                        first_bin = spectrum[:, 1]
                        power = (2 * scale) * (
                            first_bin.real.square() + first_bin.imag.square()
                        ) / hann.square().sum().clamp_min(1e-6)
                        spectral_coefficient = torch.where(
                            (complete & ~crosses_gap)[:, None], torch.log1p(power),
                            torch.zeros_like(power),
                        )
                coefficients.append(coefficient)
                spectral_coefficients.append(spectral_coefficient)
            wavelet = torch.stack(coefficients, 1)
            spectral = torch.stack(spectral_coefficients, 1)

            transition, process_noise = self._transition_and_noise(dt, dtype)
            predicted_mean = (transition @ mean[..., None]).squeeze(-1)
            propagated = transition @ covariance @ transition.transpose(-1, -2)
            prior_covariance = propagated + process_noise
            innovation = encoded[:, t] - predicted_mean[..., 0]
            innovation = torch.where(observed[:, None], innovation, torch.zeros_like(innovation))

            quality_t = torch.where(observed, qualities[:, t], torch.full_like(qualities[:, t], 0.5))
            if self.mode == "fixed":
                quality_t = torch.full_like(quality_t, 0.5)
            initial_r = base_r[None, :] * torch.exp(
                2.0 * (0.5 - quality_t[:, None]) + mode_noise[:, 1:2]
            )
            standardized = innovation / (prior_covariance[..., 0, 0] + initial_r).clamp_min(1e-6).sqrt()
            controller_input = torch.cat(
                (wavelet.flatten(1), standardized.clamp(-10.0, 10.0), previous_innovation), 1
            )
            correction = self.noise_controller(
                controller_input.to(dtype=self.noise_controller.weight.dtype)
            ).to(dtype=dtype)
            log_q, log_r = (2.0 * torch.tanh(correction)).chunk(2, dim=1)
            # A reliable, sustained innovation can mean genuine change: allow
            # larger process noise, rather than automatically rejecting it.
            alignment = (innovation * previous_innovation > 0).to(dtype=dtype)
            strength = standardized.square() / (1.0 + standardized.square())
            adaptive_process = 1.0 + 8.0 * quality_t[:, None] * strength * (0.25 + 0.75 * alignment)
            q_scale = adaptive_process * torch.exp(log_q + mode_noise[:, 0:1])
            if self.mode == "fixed":
                q_scale = torch.exp(mode_noise[:, 0:1]).expand_as(q_scale)
                log_r = torch.zeros_like(log_r)
            q_scale = torch.where(observed[:, None], q_scale, torch.exp(mode_noise[:, 0:1]))
            predicted_covariance = self._stabilize_covariance(
                propagated + process_noise * q_scale[..., None, None]
            )
            observation_noise = (initial_r * torch.exp(log_r)).clamp(1e-4, 1e4)
            innovation_variance = predicted_covariance[..., 0, 0] + observation_noise
            kalman_gain = predicted_covariance[..., :, 0] / innovation_variance[..., None]
            updated_mean = predicted_mean + kalman_gain * innovation[..., None]
            # Bound normalized latent velocity; large gaps then cause bounded
            # displacement because the velocity transition is damped.
            updated_mean = torch.stack(
                (updated_mean[..., 0], updated_mean[..., 1].clamp(-4.0, 4.0)), -1
            )

            # Joseph form protects positive semidefiniteness under roundoff.
            kh = torch.stack((kalman_gain, torch.zeros_like(kalman_gain)), -1)
            left = identity - kh
            updated_covariance = (
                left @ predicted_covariance @ left.transpose(-1, -2)
                + observation_noise[..., None, None]
                * kalman_gain[..., :, None] * kalman_gain[..., None, :]
            )
            mean = torch.where(observed[:, None, None], updated_mean, predicted_mean)
            covariance = torch.where(
                observed[:, None, None, None], updated_covariance, predicted_covariance
            )
            covariance = self._stabilize_covariance(covariance)
            previous_innovation = torch.where(observed[:, None], innovation, previous_innovation)
            states.append(mean[..., 0])
            velocities.append(mean[..., 1])
            innovations.append(innovation)
            gains.append(torch.where(observed[:, None], kalman_gain[..., 0], torch.zeros_like(innovation)))
            details.append(wavelet)
            spectra.append(spectral)

        state = torch.stack(states, 1)
        velocity = torch.stack(velocities, 1)
        innovation = torch.stack(innovations, 1)
        gain = torch.stack(gains, 1)
        wavelet = torch.stack(details, 1)
        spectral = torch.stack(spectra, 1)
        if self.mode in ("wavelet", "stft"):
            transform = spectral if self.mode == "stft" else wavelet
            summary = torch.cat((
                torch.zeros_like(state), torch.zeros_like(velocity),
                torch.zeros_like(innovation), transform.flatten(2),
            ), -1)
        elif self.mode == "fixed":
            summary = torch.cat((
                state, velocity, innovation, torch.zeros_like(wavelet.flatten(2)),
            ), -1)
        else:
            summary = torch.cat((state, velocity, innovation, wavelet.flatten(2)), -1)
        residual = self.residual_projection(
            summary.to(dtype=self.residual_projection.weight.dtype)
        ).to(dtype=features.dtype)
        use_residual = valid & (torch.arange(length, device=device)[None, :] > 0)
        residual = torch.where(use_residual[..., None], residual, torch.zeros_like(residual))
        fused = features if length == 1 else features + residual
        return {
            "features": fused,
            "state": state,
            "innovation": innovation,
            "gain": gain,
            "wavelet": wavelet,
            "spectral": spectral,
        }
