function train(npc::Int64 = 50)
    train_images = read_idx("data/train-images-idx3-ubyte")

    @assert length(size(train_images)) == 3 "invalid train_images"
    @assert 0 < size(train_images, 1) "invalid train_images"
    @assert 0 < size(train_images, 2) "invalid train_images"

    train_w, train_h, train_n = size(train_images)

    train_x = deepcopy(train_images)
    train_x = reshape(train_x, train_w*train_h, train_n)

    mean_train_x = mean(train_x; dims = 2)
    dev_train_x = train_x .- mean_train_x

    _, _, train_V = LinearAlgebra.svd(dev_train_x')

    train_V = train_V[:, 1:npc]

    train_projected_x = train_V'*train_x

    write_idx("results/train-V-idx2-ubyte", train_V)
    write_idx("results/train-projected-x-idx2-ubyte", train_projected_x)
end
