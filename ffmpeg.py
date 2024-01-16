!curl https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o ffmpeg.tar.xz
!tar -xf ffmpeg.tar.xz && rm ffmpeg.tar.xz
ffmdir = !find . -iname ffmpeg-*-static
path = %env PATH
path = path + ':' + ffmdir[0]
%env PATH $path
