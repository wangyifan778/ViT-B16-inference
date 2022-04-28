function [image] = softmax(img)
image = zeros(size(img));
    if ismatrix(img)
        [r, ~] = size(img);
        for i = 1: r
            s = sum(exp(img(i, :)));
            if s == inf
                image(i, :) = 0;
            else
            image(i, :) = exp(img(i, :)) / s;
            end
        end
    else
        image = exp(img) / sum(exp(img));
    end
end