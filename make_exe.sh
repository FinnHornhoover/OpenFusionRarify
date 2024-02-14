mkdir -p build/win_build && mkdir -p build/unix_build
docker run -v "$(pwd):/src/" cdrx/pyinstaller-windows "cd build/win_build && pyinstaller -F ../../main.py -n rarify"
cd build/unix_build && pyinstaller -F ../../main.py -n rarify && cd ../..
