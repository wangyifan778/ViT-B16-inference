clc;clear;
image_file = fopen('val.txt');
image_name = strings;
image_label = zeros(1,50000);
i = 1;
tline = fgetl(image_file);
while ischar(tline)
  image_name(i) = tline(1:28);
  image_label(i) = str2double(tline(30:end));
  tline = fgetl(image_file);
  i = i + 1;
end
fclose(image_file);

result_top1 = zeros(1,50000);
result_top5 = zeros(1,50000);

parfor img_i = 1 : 50000
    img = imread('./image/' + image_name(img_i));
    img = single(imresize(img, [384, 384]));
    img = img / 255;
    img = normalize(img, 'center', 0.5, 'scale', 0.5);
    head = 12;
    d_k = 8;
    patch = 16;
    D = 768;
    N = 577;
    
    % class_token & patch_embedding
    cls = squeeze(readNPY('./ViT-B_16/cls.npy'))';
    patch_w = readNPY('./ViT-B_16/embedding/kernel.npy');
    patch_b = permute(readNPY('./ViT-B_16/embedding/bias.npy'), [2, 1]);
    img = patch_e(img, cls, patch_w, patch_b, D, N);
    
    % pos_embeddind
    pos = squeeze(readNPY('./ViT-B_16/Transformer/posembed_input/pos_embedding.npy'));
    img = pos + img;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%blocks%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for ii = 1 : 12
        % layer_normalization1
        img1 = img;         
        norm_w = readNPY(['./ViT-B_16/Transformer/encoderblock_', num2str(ii), '/LayerNorm_0/scale.npy']);
        norm_b = readNPY(['./ViT-B_16/Transformer/encoderblock_', num2str(ii), '/LayerNorm_0/bias.npy']);
        img = layer_norm(img, norm_w, norm_b, D, N);
    
        % multi-head self-attention
        q_w = readNPY(['./ViT-B_16/Transformer/encoderblock_', num2str(ii), '/MultiHeadDotProductAttention_1/query/kernel.npy']);
        q_b = readNPY(['./ViT-B_16/Transformer/encoderblock_', num2str(ii), '/MultiHeadDotProductAttention_1/query/bias.npy']);
        k_w = readNPY(['./ViT-B_16/Transformer/encoderblock_', num2str(ii), '/MultiHeadDotProductAttention_1/key/kernel.npy']);
        k_b = readNPY(['./ViT-B_16/Transformer/encoderblock_', num2str(ii), '/MultiHeadDotProductAttention_1/key/bias.npy']);
        v_w = readNPY(['./ViT-B_16/Transformer/encoderblock_', num2str(ii), '/MultiHeadDotProductAttention_1/value/kernel.npy']);
        v_b = readNPY(['./ViT-B_16/Transformer/encoderblock_', num2str(ii), '/MultiHeadDotProductAttention_1/value/bias.npy']);
        o_w = readNPY(['./ViT-B_16/Transformer/encoderblock_', num2str(ii), '/MultiHeadDotProductAttention_1/out/kernel.npy']);
        o_b = readNPY(['./ViT-B_16/Transformer/encoderblock_', num2str(ii), '/MultiHeadDotProductAttention_1/out/bias.npy']);
        img = msa(img, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b, head, d_k, D, N);
        img = img1 + img;
    
        % multilayer perceptron
        img2 = img;
        norm_w = readNPY(['./ViT-B_16/Transformer/encoderblock_', num2str(ii), '/LayerNorm_2/scale.npy']);
        norm_b = readNPY(['./ViT-B_16/Transformer/encoderblock_', num2str(ii), '/LayerNorm_2/bias.npy']);
        fc1_w = readNPY(['./ViT-B_16/Transformer/encoderblock_', num2str(ii), '/MlpBlock_3/Dense_0/kernel.npy']);
        fc1_b = permute(readNPY(['./ViT-B_16/Transformer/encoderblock_', num2str(ii), '/MlpBlock_3/Dense_0/bias.npy']), [2, 1]);
        fc2_w = readNPY(['./ViT-B_16/Transformer/encoderblock_', num2str(ii), '/MlpBlock_3/Dense_1/kernel.npy']);
        fc2_b = permute(readNPY(['./ViT-B_16/Transformer/encoderblock_', num2str(ii), '/MlpBlock_3/Dense_1/bias.npy']), [2, 1]);
        img = mlp(img, norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b, D, N);
        img = img2 + img;
     end
    
     % layer_normalization
     norm_w = readNPY('./ViT-B_16/Transformer/encoder_norm/scale.npy');
     norm_b = readNPY('./ViT-B_16/Transformer/encoder_norm/bias.npy');
     img = layer_norm(img, norm_w, norm_b, D, N);
    
     % mlp head
     fc_w = readNPY('./ViT-B_16/head/kernel.npy');
     fc_b =permute(readNPY('./ViT-B_16/head/bias.npy'), [2, 1]);
     img = full_connected(img(1, :), fc_w, fc_b);
     img = softmax(img);
    
     %TOP1
     [pred_acc,pred_label] = max(img);
     if pred_label - 1 == image_label(img_i)
       result_top1(img_i) = 1;
     end
      
     %TOP5
     [b,pred_label_top5]=sort(img,'descend');
     pred_label_top5 = pred_label_top5 -1;
     if find(pred_label_top5(1:5) == image_label(img_i))
       result_top5(img_i) = 1;
     else
       result_top5(img_i) = 0;  
     end
end

%TOP1_accuracy
top1_acc = sum(result_top1) / 50000;

%TOP5_accuracy
top5_acc = sum(result_top5) / 50000;

