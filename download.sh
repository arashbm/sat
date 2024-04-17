for version in random block culture_10; do
  for split in training testing; do
    wget -P data/$version/ ftp://m1613658:m1613658@dataserv.ub.tum.de/$version/$split.h5
  done
done
