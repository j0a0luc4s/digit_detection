function test(npc::Int64 = 50)
    t10k_images = read_idx("data/t10k-images-idx3-ubyte")
    t10k_labels = read_idx("data/t10k-labels-idx1-ubyte")

    @assert length(size(t10k_images)) == 3 "invalid t10k_images"
    @assert length(size(t10k_labels)) == 1 "invalid t10k_labels"
    @assert 0 < size(t10k_images, 1) "invalid t10k_images"
    @assert 0 < size(t10k_images, 2) "invalid t10k_images"
    @assert size(t10k_images, 3) == size(t10k_labels, 1) "t10k_images and t10k_labels dimension mismatch"

    t10k_w, t10k_h, t10k_n = size(t10k_images)

    t10k_x = deepcopy(t10k_images)
    t10k_x = reshape(t10k_x, t10k_w*t10k_h, t10k_n)

    train_V = read_idx("results/train-V-idx2-ubyte")
    train_projected_x = read_idx("results/train-projected-x-idx2-ubyte")
    train_labels = read_idx("data/train-labels-idx1-ubyte")

    @assert length(size(train_V)) == 2 "invalid train_V"
    @assert length(size(train_projected_x)) == 2 "invalid train_projected_x"
    @assert length(size(train_labels)) == 1 "invalid train_labels"
    @assert size(train_V) == (t10k_w*t10k_h, npc) "invalid train_V"
    @assert size(train_projected_x, 1) == npc "invalid train_projected_x"
    @assert 0 < size(train_projected_x, 2) "invalid train_projected_x"
    @assert size(train_projected_x, 2) == size(train_labels, 1) "train_labels and train_projected_x dimension mismatch"

    t10k_projected_x = train_V'*t10k_x

    t10k_test_labels = eltype(t10k_labels)[]
    t10k_test_idxs = Int32[]

    for t10k_projected_x_col in eachcol(t10k_projected_x)
        norms = [
            norm(col)
            for col in eachcol(train_projected_x .- t10k_projected_x_col)
        ]
        idx = findmin(norms)[2]
        push!(t10k_test_labels, train_labels[idx])
        push!(t10k_test_idxs, idx)
    end

    write_idx("results/t10k-projected-x-idx2-ubyte", t10k_projected_x)
    write_idx("results/t10k-test-labels-idx1-ubyte", t10k_test_labels)
    write_idx("results/t10k-test-idxs-idx2-ubyte", t10k_test_idxs)
end
