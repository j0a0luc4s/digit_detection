function showcase(npc::Int64 = 50)
    n_projection_samples = 3
    n_match_samples = 3
    n_mismatch_samples = 3

    train_images      = read_idx("data/train-images-idx3-ubyte")
    train_labels      = read_idx("data/train-labels-idx1-ubyte")
    train_projected_x = read_idx("results/train-projected-x-idx2-ubyte")
    train_V           = read_idx("results/train-V-idx2-ubyte")
    t10k_images       = read_idx("data/t10k-images-idx3-ubyte")
    t10k_labels       = read_idx("data/t10k-labels-idx1-ubyte")
    t10k_projected_x  = read_idx("results/t10k-projected-x-idx2-ubyte")
    t10k_test_labels  = read_idx("results/t10k-test-labels-idx1-ubyte")
    t10k_test_idxs    = read_idx("results/t10k-test-idxs-idx2-ubyte")

    @assert length(size(train_images))      == 3 "invalid train_images"
    @assert length(size(train_labels))      == 1 "invalid train_labels"
    @assert length(size(train_projected_x)) == 2 "invalid train_projected_x"
    @assert length(size(train_V))           == 2 "invalid train_V"
    @assert length(size(t10k_images))       == 3 "invalid t10k_images"
    @assert length(size(t10k_labels))       == 1 "invalid t10k_labels"
    @assert length(size(t10k_projected_x))  == 2 "invalid t10k_projected_x"
    @assert length(size(t10k_test_labels))  == 1 "invalid t10k_test_labels"
    @assert length(size(t10k_test_idxs))    == 1 "invalid t10k_test_idxs"

    @assert 0 < size(train_images, 1) ==
                size(t10k_images, 1)  && 
            0 < size(train_images, 2) ==
                size(t10k_images, 2) "invalid images"
    @assert size(train_projected_x, 1) ==
            size(t10k_projected_x, 1)  ==
            npc "invalid projected_x"
    @assert size(train_V) ==
            (size(train_images, 1)*size(train_images, 2), npc) "invalid V"
    @assert size(train_images, 3) ==
            size(train_labels, 1) ==
            size(train_projected_x, 2) "dimension mismatch"
    @assert size(t10k_images, 3)      ==
            size(t10k_labels, 1)      ==
            size(t10k_test_labels, 1) ==
            size(t10k_test_idxs, 1)   ==
            size(t10k_projected_x, 2) "dimension mismatch"

    projection_samples = sample(
        1:size(t10k_images, 3),
        n_projection_samples; replace=false)

    for projection_sample in projection_samples
        plot(
            heatmap(
                reverse(
                    reshape(
                        train_V*t10k_projected_x[:, projection_sample],
                        (size(train_images, 1), size(train_images, 2))
                    )';
                    dims=1
                );
                title = "Projected t10k image"
            ),
            heatmap(
                reverse(
                    t10k_images[:, :, projection_sample]';
                    dims=1
                );
                title = "Actual t10k image"
            );
        )
        savefig("projection_" * string(projection_sample))
    end

    match_samples = sample(
        findall(t10k_labels .== t10k_test_labels),
        n_match_samples; replace=false)

    for match_sample in match_samples
        plot(
            heatmap(
                reverse(
                    train_images[:, :, t10k_test_idxs[match_sample]]';
                    dims=1
                );
                title = "Closest train image: " * string(
                    train_labels[t10k_test_idxs[match_sample]]
                )
            ),
            heatmap(
                reverse(
                    t10k_images[:, :, match_sample]';
                    dims=1
                );
                title = "Actual t10k image: " * string(
                    t10k_labels[match_sample]
                )
            );
        )
        savefig("match_" * string(match_sample))
    end

    mismatch_samples = sample(
        findall(t10k_labels .!= t10k_test_labels),
        n_mismatch_samples; replace=false)

    for mismatch_sample in mismatch_samples
        plot(
            heatmap(
                reverse(
                    train_images[:, :, t10k_test_idxs[mismatch_sample]]';
                    dims=1
                );
                title = "Closest train image: " * string(
                    train_labels[t10k_test_idxs[mismatch_sample]]
                )
            ),
            heatmap(
                reverse(
                    t10k_images[:, :, mismatch_sample]';
                    dims=1
                );
                title = "Actual t10k image: " * string(
                    t10k_labels[mismatch_sample]
                )
            );
        )
        savefig("mismatch_" * string(mismatch_sample))
    end
end
