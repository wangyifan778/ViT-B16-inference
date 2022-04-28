function [image] = full_connected(img, fc_w, fc_b)
    image = img * fc_w + fc_b;
end