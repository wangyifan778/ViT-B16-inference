function [image] = softmax(img)
image = zeros(size(img));
    if ismatrix(img)
        [r, ~] = size(img);
        for i = 1: r
            image(i, :) = exp(img(i, :) - max(img(i, :))) / sum(exp(img(i, :) - max(img(i, :))));
        end
    else
        image = exp(img - max(img)) / sum(exp(img - max(img)));
    end
end
