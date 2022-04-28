function [image] = gelu(img)
    image = 0.5 * img .* (1 + tanh(sqrt(2 / pi) * (img + 0.044715 * img .^ 3)));
end