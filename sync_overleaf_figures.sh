cd overleaf
git pull --rebase
cp ../figures/*.png figures/
git add figures/*.png
git commit -m "Sync figures"
git push
