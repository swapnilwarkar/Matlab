function [Process_Image, Mask_Image] = findshadow(opt)

    if ~isfield(opt, 'pairwiseonly')
       opt.pairwiseonly = 0;
    end  
    
    basename = opt.fn(1:end-4);
    im       = imread([opt.dir 'original/' opt.fn]);
        
    oim = im;
    org_siz = [size(im, 1) size(im, 2)];
    
    resiz_ratio = 640/org_siz(2);
        
    im = imresize(oim, resiz_ratio);
    
    disp 'Step 1 : Segmentation of the Image.'
    load([opt.dir 'cache/' basename '_seg.mat']);
    numlabel = length(unique(seg(:)));
    
    load([opt.dir 'cache/' basename '_single.mat']);
    load(opt.unaryClassifier, 'model');
    disp 'Single region SVM classification'
    
    disp 'Step 2 : SVM(Support Vector Machine) with Binary Classifier with Linear SVM.'
    [err, s]=svmclassify(testdata, testlabel, model);
    ss = double(sign(s));
    ssmap=(1-ss(seg))/2;
        
    load([opt.dir 'cache/' basename '_pair.mat']);
    load(opt.binaryClassifier, 'diffmodel', 'samemodel');
    
    disp 'Pair region SVM classification'
    [err, s1]=svmclassify(finalvector, zeros(numlabel*numlabel, 1) , diffmodel);
    [err, s2]=svmclassify(finalvector, zeros(numlabel*numlabel, 1), samemodel);  
    
    s1 = reshape(s1, [numlabel numlabel]);
    s2 = reshape(s2, [numlabel numlabel]);
    k1=100; k2=200;
    t1 = sort(s1(:));t1(isnan(t1))=[];
    t2 = sort(s2(:));t2(isnan(t2))=[];
    thresh1 = t1(max(1, length(t1(:))-k1)); thresh1 = max(.6, thresh1);
    thresh2 = t2(max(1, length(t2(:))-k2)); thresh2 = max(.6, thresh2);
    
    if strcmp('models/model_pair_our.mat', opt.binaryClassifier)
    for i=1:numlabel
        for j=1:numlabel
            if i==j , continue; end
            w = sqrt(shapemean.area(i)*shapemean.area(j));
            g_diff(i,j) = w*s1(j,i);
            g_same(i,j) = w*s2(j,i);
        end
    end
    else
    for i=1:numlabel
        for j=1:numlabel
            if i==j , continue; end
            w = sqrt(shapemean.area(i)*shapemean.area(j));
            g_diff(i,j) = w*s1(i,j);
            g_same(i,j) = w*s2(i,j);
        end
    end
    end
    
    nNodes = numlabel;
    nStates = 2;
    adj1 = logical(sparse(nNodes,nNodes));
    adj2 = logical(sparse(nNodes,nNodes));
    
    m=buildAdjacentMatrix(seg, numlabel);
    
    for i=1:numlabel
        for j=1:numlabel
            if s1(i,j)>thresh1
                adj1(i,j)=1;
            end
            if s2(i,j)>thresh2
                adj2(i,j)=1;
            end
        end
    end
    
    nodePot = zeros(nNodes,nStates);
    w1 = 1;
    if ~opt.pairwiseonly
        for i=1:numlabel
            wi = w1 * shapemean.area(i);
            nodePot(i,1) = -s(i)*wi;
            nodePot(i,2) = s(i)*wi;
        end
    end
    
    if 1
        sc = shapemean.center;
        nim = im;
        [gx gy] = gradient(double(seg));
        eim = (gx.^2+gy.^2)>1e-10;
        
        t = nim(:,:,1); t(eim)=0; nim(:,:,1)=t;
        t = nim(:,:,2); t(eim)=0; nim(:,:,2)=t;
        t = nim(:,:,3); t(eim)=0; nim(:,:,3)=t;
    end
    
    Process_Image = nim;
    
    disp 'Step 3 : Feature Extraction with Masking'
    edgeStruct1 = UGM_makeEdgeStruct_directed(adj1,nStates);
    edgeStruct2 = UGM_makeEdgeStruct_directed(adj2,nStates);
    edgePot = [];
    w3=1;
    w2=2;
    for e = 1:edgeStruct1.nEdges
        n1 = edgeStruct1.edgeEnds(e,1);
        n2 = edgeStruct1.edgeEnds(e,2);
        nodePot(n1,1) = nodePot(n1,1)+ w2*g_diff(n1, n2);
        nodePot(n1,2) = nodePot(n1,2)- w2*g_diff(n1, n2);
        nodePot(n2,1) = nodePot(n2,1)- w2*g_diff(n1, n2);
        nodePot(n2,2) = nodePot(n2,2)+ w2*g_diff(n1, n2);
    end
        
    for e = 1:edgeStruct2.nEdges
        n1 = edgeStruct2.edgeEnds(e,1);
        n2 = edgeStruct2.edgeEnds(e,2);

        edgePot(:,:,e) = [g_same(n1, n2) 0;...
            0, g_same(n1, n2) ].*[w3 1; 1 w3];
    end
    
    if ~isempty(edgePot)
        Decoding = UGM_Decode_ModifiedCut(nodePot,edgePot,edgeStruct2);
    else
        Decoding = double(sign(s))+1;
    end
    
    hardmap = Decoding(seg)-1;
    
    imwrite(hardmap,[opt.dir 'binary/' basename '_binary.png']);	
    ss = double(sign(s));
    ssmap=(1-ss(seg))/2;       
    imwrite(1-ssmap,[opt.dir 'unary/' basename '_unary.png']);
    save([opt.dir 'cache/' basename '_detect.mat'],'hardmap','ssmap');
        
    shadowim = hardmap;
    tmp = ones(size(im));
    tmp2 = cat(3, zeros([size(shadowim) 2]), 0.5*shadowim);
    mask = logical(repmat(shadowim, [1 1 3]));
    tmp(mask) = tmp2(mask);
    
    grayim = im2double(repmat(rgb2gray(im), [1 1 3]));
    im2 = (grayim+tmp)/2;
    imwrite(im2, [opt.dir 'mask/' basename '_mask.png']);
    
    %figure(3), imshow(im2); title('Mask IMAGE');
    Mask_Image = im2;
    
    RegionScore = s;
    DiffScore = s1;
    SameScore = s2;
    DiffAdj = adj1;
    SameAdj = adj2;
    save([opt.dir 'cache/' basename '_everything.mat'], 'RegionScore', 'DiffScore', 'SameScore', 'DiffAdj', 'SameAdj', 'seg', 'im', 'hardmap', 'ssmap');
    
    % get the pairing information consistent with the final detection.
    [shadow, non_shadow] = find(adj1 == 1);
    
    pair = [];
    
    num_pair = numel(non_shadow);
    for i = 1:num_pair
       if Decoding(non_shadow(i))==1 && Decoding(shadow(i)) == 2
           pair = [pair; non_shadow(i) shadow(i)];
       end
    end
   
    save([opt.dir 'cache/' basename '_finalpair.mat'], 'pair');
end
