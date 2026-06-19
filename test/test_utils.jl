using PhotoAcoustic
using LinearAlgebra
using Test
using DSP: blackman
using FFTW: fft, ifft, ifftshift
import FourierTools

@testset "utils" begin

    @testset "circle_geometry 2D" begin
        cx, cy, r, n = 1f0, 2f0, 0.5f0, 64
        xs, ys, theta = circle_geometry((cx, cy), r, n)
        @test length(xs) == n
        @test length(ys) == n
        @test length(theta) == n
        @test eltype(xs) == Float32
        @test eltype(ys) == Float32
        @test eltype(theta) == Float32
        # every point exactly r from the centre
        radii = sqrt.((xs .- cx).^2 .+ (ys .- cy).^2)
        @test all(isapprox.(radii, r; atol=1f-5))
        # endpoints of the angle sweep
        @test theta[1] == 0f0
        @test theta[end] < 2f0*pi
    end

    @testset "circle_geometry 3D" begin
        c = (1f0, 2f0, 3f0)
        r, n = 0.5f0, 32
        xs, ys, zs, (theta, phi) = circle_geometry(c, r, n)
        @test length(xs) == n * n
        @test length(ys) == n * n
        @test length(zs) == n * n
        @test length(theta) == n
        @test length(phi) == n
        # every point exactly r from the centre (allow Float32 roundoff)
        r2 = (xs .- c[1]).^2 .+ (ys .- c[2]).^2 .+ (zs .- c[3]).^2
        @test all(isapprox.(r2, r^2; atol=1f-3))
    end

    @testset "pad_zeros / unpad round-trip" begin
        # 1-D
        m = Float32.(collect(1:10))
        nb = ((2, 3),)
        p = PhotoAcoustic.pad_zeros(m, nb)
        @test size(p) == (15,)
        @test all(p[1:2] .== 0)
        @test p[3:12] == m
        @test all(p[13:15] .== 0)
        # 2-D, asymmetric
        m2 = Float32.(reshape(collect(1:12), 3, 4))
        nb2 = ((1, 2), (3, 1))
        p2 = PhotoAcoustic.pad_zeros(m2, nb2)
        @test size(p2) == (3 + 1 + 2, 4 + 3 + 1)
        @test p2[2:4, 4:7] == m2
        # exact round-trip with symmetric padding
        m3 = randn(Float32, 5, 7)
        nb3 = ((2, 2), (3, 3))
        @test PhotoAcoustic.unpad(PhotoAcoustic.pad_zeros(m3, nb3), nb3) ≈ m3
    end

    @testset "pad_zeros / unpad adjoint (dottest)" begin
        m = randn(Float32, 6, 8)
        nb = ((1, 2), (3, 4))
        Pm = PhotoAcoustic.pad_zeros(m, nb)
        y = randn(Float32, size(Pm))
        Pty = PhotoAcoustic.unpad(y, nb)
        lhs = dot(vec(Pm), vec(y))
        rhs = dot(vec(m), vec(Pty))
        @test isapprox(lhs, rhs; rtol=1f-4)
    end

    @testset "blackman_upscale: square output contract" begin
        n = (32, 32)
        d = (0.1f0, 0.1f0)
        p0 = zeros(Float32, n)
        p0[div(n[1], 2), div(n[2], 2)] = 1f0
        out, d_up = blackman_upscale(p0, d)
        # output is close to n * upsample_fact on each axis (within one cell —
        # exact value depends on how the physical extent rounds through ceil
        # before the boundary padding is removed)
        @test all(abs.(size(out) .- n .* 1.25) .<= 1)
        @test d_up == Float32.(d ./ 1.25f0)
        @test maximum(abs.(out)) ≈ 1f0
        @test all(isfinite.(out))
    end

    @testset "blackman_upscale: bitwise match with the pre-fix (symmetric) path on square grids" begin
        # On square inputs where (N_up - N_orig) is even on every axis, the new
        # asymmetric padding `(diff÷2, diff - diff÷2)` collapses to the old
        # symmetric `(diff÷2, diff÷2)`, so the output must equal the original
        # algorithm's output bit-for-bit. The reference below reproduces what
        # the pre-fix code does, mirroring src/utils.jl as it was prior to the
        # asymmetric-padding fix.
        function blackman_upscale_old(p0::Array{T, N}, d::NTuple{N, T},
                                      upsample_fact=1.25; pad_a=16) where {T, N}
            try
                Int(upsample_fact * pad_a)
            catch InexactError
                throw(ArgumentError("Padding must be integer after upsampling"))
            end
            pad = ntuple(_ -> (pad_a, pad_a), N)
            p0_zeropad = PhotoAcoustic.pad_zeros(p0, pad)
            N_orig = size(p0_zeropad)
            x       = d .* (N_orig .- 1)
            d_up    = Float32.(d ./ upsample_fact)
            N_up    = ceil.(Int, x ./ d_up)
            p0_up   = FourierTools.resample(p0_zeropad, N_up; normalize=true)
            window  = Float32.(blackman(N_orig; padding=0))
            pad     = (N_up .- N_orig) .÷ 2
            pad     = ntuple(i -> (pad[i], pad[i]), N)       # OLD: symmetric
            window_padded = PhotoAcoustic.pad_zeros(window, pad)
            p0_up_smooth = real.(ifft(fft(p0_up) .* ifftshift(window_padded)))
            p0_up_smooth = p0_up_smooth / norm(p0_up_smooth, Inf)
            zero_pad = Int(upsample_fact * pad_a)
            zpad     = ntuple(_ -> (zero_pad, zero_pad), N)
            p0_up_smooth = PhotoAcoustic.unpad(p0_up_smooth, zpad)
            return p0_up_smooth, d_up
        end

        # Pick a square shape where diff is even on both axes under the defaults
        # — that is the only regime the pre-fix code did not crash on, so it is the
        # only regime where "produces same result as original" is well-defined.
        n  = (50, 50)
        d  = (0.1f0, 0.1f0)
        p0 = randn(Float32, n)

        # confirm the assumption: diff is even on both axes for this case
        N_orig = n .+ (32, 32)              # 2 * pad_a per axis
        x      = d .* (N_orig .- 1)
        d_up   = Float32.(d ./ 1.25f0)
        N_up   = ceil.(Int, x ./ d_up)
        @test all(iseven.(N_up .- N_orig))

        new_out, _ = blackman_upscale(p0, d)
        old_out, _ = blackman_upscale_old(p0, d)
        @test new_out == old_out
    end

    @testset "blackman_upscale: rectangular grid" begin
        # Shape chosen so that (N_up - N_orig) has one odd axis under the
        # defaults — this is the originally-broken case.
        n  = (24, 36)
        d  = (0.1f0, 0.05f0)
        p0 = zeros(Float32, n)
        p0[div(n[1], 2), div(n[2], 2)] = 1f0

        # sanity-check the test case actually exercises the asymmetric path
        N_orig = n .+ (32, 32)
        x      = d .* (N_orig .- 1)
        d_up   = Float32.(d ./ 1.25f0)
        N_up   = ceil.(Int, x ./ d_up)
        @test any(isodd.(N_up .- N_orig))

        out, d_up_actual = blackman_upscale(p0, d)
        @test all(abs.(size(out) .- n .* 1.25) .<= 1)
        @test d_up_actual == Float32.(d ./ 1.25f0)
        @test maximum(abs.(out)) ≈ 1f0
        @test all(isfinite.(out))

        # Flip which axis carries the odd diff
        n2  = (40, 24)
        d2  = (0.05f0, 0.1f0)
        p02 = zeros(Float32, n2)
        p02[div(n2[1], 2), div(n2[2], 2)] = 1f0
        out2, _ = blackman_upscale(p02, d2)
        @test all(abs.(size(out2) .- n2 .* 1.25) .<= 1)
        @test maximum(abs.(out2)) ≈ 1f0
        @test all(isfinite.(out2))
    end
end
