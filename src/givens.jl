function givens(
    a::Tp,
    b::Tp;
    atol::Tp = 1e-6
) where {Tp <: AbstractFloat}
    if isapprox(b, Tp(0.0); atol=atol)
        s = Tp(0.0)
        c = Tp(1.0)
    elseif abs(b) > abs(a)
        τ = Tp(-a/b)
        s = Tp(1.0/sqrt(1.0 + τ*τ))
        c = Tp(s*τ)
    else
        τ = Tp(-b/a)
        c = Tp(1.0/sqrt(1.0 + τ*τ))
        s = Tp(c*τ)
    end

    return Tp[c  s;
              -s c]
end
