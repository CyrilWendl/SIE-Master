for method in df net_msr net_entropy; do
convert ZH_wo_cl_1_${method}_im_3.pdf -stroke blue -strokewidth 2 -fill "rgba( 65, 105, 255, 0.5 )" -draw "rectangle 390,180 575,330 " detail_1/df_im_3_border_${method}.jpg;
done;
