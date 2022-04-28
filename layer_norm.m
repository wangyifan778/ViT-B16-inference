function [image] = layer_norm(img, norm_w, norm_b, D, N)
    norm_w = permute(norm_w, [2, 1]);
    norm_b = permute(norm_b, [2, 1]);
    image = zeros(N, D);
    eps = 1e-6;
    for i = 1 : N
        u = mean(img(i, :));
        v = var(img(i, :));
        image(i, :) = (img(i, :) - u) / sqrt(v + eps);
        image(i, :) = image(i, :) .* norm_w + norm_b;
    end
end
