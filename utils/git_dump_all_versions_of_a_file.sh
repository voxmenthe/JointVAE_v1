function show_help()
{
  echo "dump all git versions of a file to separate files"
  echo 
  echo "usage: FILENAME";
  echo 
  echo "e.g."
  echo 
  echo "$ git_dump_all_versions_of_a_file.sh path/to/somefile.txt"
  echo 
  echo "path/to/somefile.txt.1.0dea419"
  echo "path/to/somefile.txt.1.0dea419.logmsg"
  echo "path/to/somefile.txt.2.cdea8s9"
  echo "path/to/somefile.txt.2.cdea8s9.logmsg"
  echo "path/to/somefile.txt.3.fdsf2d"
  echo "path/to/somefile.txt.3.fdsf2d.logmsg"
  echo "..."
  exit;
}

if [ "$1" == "help" ]
then
  show_help
fi
if [ -z "$1" ]
then
  show_help
fi

ROOT_FOLDER=$(git rev-parse --show-toplevel)
CURR_DIR=$(pwd)
if [ "$ROOT_FOLDER" != "$CURR_DIR" ]
then
  echo "Switch to the root of the repo and try again. Should be in $ROOT_FOLDER"
  exit
fi

function git_log_message_for_commit
{
  IT=$(git log -1 --pretty=format:"%an, %s, %b, %ai"  $*)
  echo $IT
}

function git_log_short
{
  git log --oneline --decorate $*
}

function choose_col
{
  COL=$1
  if [ -z "$2" ]
  then
    OPTS=
  else
    OPTS="-F\\${2}"
  fi
  awk $OPTS -v col="$COL" '{print $col}' 
}

cd $ROOT_FOLDER
FILENAME=$*
HASHES=$(git_log_short $FILENAME | choose_col 1)
INDEX=1

for HASH in $HASHES
do
  INDEX_OUT=$(printf %03d $INDEX)
  OUT_FILENAME="$FILENAME.$INDEX_OUT.$HASH"
  OUT_LOG_FILENAME="$FILENAME.$INDEX_OUT.$HASH.logmsg"
  echo "saving version $INDEX to file $OUT_FILENAME for hash:$HASH"
  echo "*******************************************************" >>  $OUT_LOG_FILENAME
  git_log_message_for_commit $HASH >> $OUT_LOG_FILENAME
  echo "  $HASH:$FILENAME " >> $OUT_LOG_FILENAME
  echo "*******************************************************" >> $OUT_LOG_FILENAME

  git show $HASH:$FILENAME >> $OUT_FILENAME
  let INDEX=INDEX+1
done