export LPCUSER="chuyuanl"
export WFS="classifier/config/workflows/HCR/FvT_study_model"

# check port
if [ -z "$1" ]; then
    port=10200
else
    port=$1
fi

# evaluate
for i in {0..14}
do
    ./pyml.py \
    template "{mixed: ${i}, user: ${LPCUSER}}" $WFS/cache.yml \
    -setting Monitor "address: :${port}"
done
