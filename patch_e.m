function [image] = patch_e(img, cls, patch_w, patch_b, D, N)
    image = zeros(N, D);
    image(1, :)= cls;
    patch_w = reshape(permute(patch_w, [2, 1, 3, 4]), [256 3 768]);
    patch_w = permute(reshape(patch_w, [768 768]), [1, 2]);
    patch_b = repmat(patch_b, 576, 1);
    dim = ones(1, 24) * 16;
    img = permute(img, [2, 1, 3]);
    img = mat2cell(img, dim, dim, 3);
    for i = 1 : N-1
        image(i+1, :) = reshape(img{i},[1 768]);
    end
    image(2 : N, :) = image(2 : N, :) * patch_w + patch_b;
%     for i = 1 : N-1
%         for j = 1 : D
%             for c = 1 : 3
%                 image(i+1, j) = image(i+1, j) + sum(permute(img{i}(:, :, c), [2, 1, 3]) .* patch_w(:, :, c, j), "all");
%             end
%             image(i+1, j) = image(i+1, j) + patch_b(j);
%         end
%     end
end
