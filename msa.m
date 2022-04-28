function [image1] = msa(img, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b, head, d_k, D, N)
    image = zeros(N, D);
    image1 = zeros(N, D);
    q_w = permute(q_w, [1, 3, 2]);
    k_w = permute(k_w, [1, 3, 2]);
    v_w = permute(v_w, [1, 3, 2]);
    o_w = permute(o_w, [2, 3, 1]);
    for i = 1 : head
        Q = img * q_w(:, :, i) + repmat(q_b(i, :), 577, 1);
        K = img * k_w(:, :, i) + repmat(k_b(i, :), 577, 1);
        V = img * v_w(:, :, i) + repmat(v_b(i, :), 577, 1);
        Z = Q * K' / d_k;
        image(:, 64 * (i - 1) + 1 : 64 * i) = softmax(Z) * V;
        image1 = image1 + image(:, 64 * (i - 1) + 1 : 64 * i) * o_w(:, :, i);
    end
    image1 = image1 + permute(o_b, [2, 1]);
end