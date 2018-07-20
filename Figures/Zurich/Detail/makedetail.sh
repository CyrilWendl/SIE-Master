CL=3
IM=3
for METHOD in df; do
    convert  ../Im_cert/cl_${CL}/ZH_wo_cl_${CL}_${METHOD}_im_${IM}.jpg bz.png
    convert bz.png\
    \( +clone -blur 0x5 \)  \
    \( +clone -fill White -colorize 100 -fill Black -draw "rectangle 100,100,499,149" \) \
    -composite bz1.png
done;
