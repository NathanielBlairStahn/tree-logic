#Replaces spaces with underscores in filenames, thanks to Stack Exchange:
#https://stackoverflow.com/questions/11752878/awk-sed-one-liner-command-for-removing-spaces-from-all-file-names-in-a-given
#https://unix.stackexchange.com/questions/223182/how-to-replace-spaces-in-all-file-names-with-underscore-in-linux-using-shell-scr
for name in *\ *; do mv -v "$name" "${name// /_}"; done

#Example:
#image_douglas fir_99.png -> image_douglas_fir_99.png
