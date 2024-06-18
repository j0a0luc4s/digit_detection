const types = Dict(
    0x08 => UInt8,
    0x09 => Int8,
    0x0B => Int16,
    0x0C => Int32,
    0x0D => Float32,
    0x0E => Float64,
)


function read_idx(filename::AbstractString)
    file = open(filename)

    @assert read(file, Int8) == 0 "invalid"
    @assert read(file, Int8) == 0 "invalid"

    typebyte = read(file, Int8)
    @assert typebyte in keys(types) "invalid"
    type = types[typebyte]

    numdims = read(file, Int8)
    @assert 0 < numdims "invalid"
    dims = UInt32[]
    for _ in 1:numdims
        push!(dims, bswap(read(file, UInt32)))
        @assert 0 < dims[end] "invalid"
    end

    raw = reinterpret(type, read(file))
    data = reshape(raw, tuple(reverse(dims)...))

    close(file)

    return data
end

const typebytes = Dict(
    UInt8   => 0x08,
    Int8    => 0x09,
    Int16   => 0x0B,
    Int32   => 0x0C,
    Float32 => 0x0D,
    Float64 => 0x0E,
)

function write_idx(filename::AbstractString, data::AbstractArray{type, numdims}) where {type, numdims}
    file = open(filename, "w")

    write(file, Int8(0))
    write(file, Int8(0))

    @assert type in keys(typebytes) "invalid"
    typebyte = typebytes[type]
    write(file, Int8(typebyte))

    @assert 0 < numdims "invalid"
    write(file, Int8(numdims))
    dims = size(data)
    for dim in reverse(dims)
        @assert 0 < dim "invalid"
        write(file, bswap(UInt32(dim)))
    end

    write(file, reinterpret(UInt8, data))

    close(file)
end
