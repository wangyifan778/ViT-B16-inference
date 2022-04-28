function [image] = mlp(img, norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b, D, N)
    fc1_b = repmat(fc1_b, N, 1);
    fc2_b = repmat(fc2_b, N, 1);
    image = layer_norm(img, norm_w, norm_b, D, N);
    image = full_connected(image, fc1_w, fc1_b);
    image = gelu(image);
    image = full_connected(image, fc2_w, fc2_b);
end