
f_link_frames () {
    FRAME_PATH="$1"
    LABEL_PATH="$2"

    for label in "$LABEL_PATH"/*.txt; do
        if [ -f "$label" ]; then
            image="${label%.txt}.png"        # change extension
            image="$FRAME_PATH/${image##*/}" # keep fname w/o path
            ln -sv "$image" "$LABEL_PATH/"
        fi
    done
}


f_unlink_frames () {
    UNLINK_PATH="${1:=.}"

    find "$UNLINK_PATH" -type l -exec rm {} \;
}
