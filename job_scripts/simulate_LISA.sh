#REMARK: run as $ . lisa_local.sh instead of $ ./lisa_local.sh
# create env var that points to empty dir (LISA: local file system)
export HOME="/home/maartje/Documents/UvA-Courses/SummerProject/LISA"
export TMPDIR=$HOME/"TMP"
rm -rf $TMPDIR
mkdir $TMPDIR
