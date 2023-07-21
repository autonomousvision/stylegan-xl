#! /bin/bash
dir_to_sync='VideoGAN'
usr=$1
host_machine=$(hostname)
while inotifywait -r --exclude '/\.' ../$dir_to_sync/*; do
  if [ "$host_machine" = "brown" ]; then
    rsync --exclude=".*" -av ../$dir_to_sync/ /is/cluster/$usr/repos/never_eidt_from_cluster_remote_edit_loc/$dir_to_sync &
    sleep 2
    echo "second sync."
    rsync --exclude=".*" -av ../$dir_to_sync/ /is/cluster/$usr/repos/never_eidt_from_cluster_remote_edit_loc/$dir_to_sync
  else
    rsync --exclude=".*" -azhe "ssh -i ~/.ssh/id_rsa" ../$dir_to_sync/ $usr@brown.is.localnet:/is/cluster/$usr/repos/never_eidt_from_cluster_remote_edit_loc/$dir_to_sync &
    sleep 2
    echo "second sync."
    rsync --exclude=".*" -azhe "ssh -i ~/.ssh/id_rsa" ../$dir_to_sync/ $usr@brown.is.localnet:/is/cluster/$usr/repos/never_eidt_from_cluster_remote_edit_loc/$dir_to_sync
  fi
done
