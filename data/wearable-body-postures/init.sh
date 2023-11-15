#/bin/bash

wget http://groupware.les.inf.puc-rio.br/static/har/dataset-har-PUC-Rio-ugulino.zip
unzip dataset-har-PUC-Rio-ugulino.zip

echo "The original dataset contains a typo on line 122078." 
echo "It seems, that the current date has been pasted into this line. This was changed as the following:"
echo "-14420-11-2011 04:50:23.713 -> -144