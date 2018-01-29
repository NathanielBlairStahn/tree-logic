#Replaces spaces with underscores in filenames
for name in *\ *; do mv -v "$name" "${name// /_}"; done

#Example:
#image_douglas fir_99.png -> image_douglas_fir_99.png
