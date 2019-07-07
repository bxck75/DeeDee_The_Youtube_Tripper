#! /bin/bash
# ./trip_that_video.sh https://www.youtube.com/watch?v=qmsbP13xu6k flatbeat.mp4 134
echo "Setting up shop"
rip_url=$1
filename=$2
download_folder='downloads/'
name=`echo $2|awk -F '.' '{print $1}'`
extension='jpg'
quality=$3
rip=$4
if [ "${rip}" == "rip" ]; then

	echo "Making folders"
	mkdir -p ./frames/$name
	mkdir -p ./processed_frames/$name
	# delete old video 
	rm $download_folder$filename

	echo "Ripping Video from Youtube"
	youtube-dl -f $quality --output $download_folder$filename $rip_url

	echo "Splitting the mp4 file to jpg frames"
	./1_movie2frames.sh ffmpeg $download_folder$filename ./frames/$name $extension
fi

echo "DeepDreaming the frames (This is a long task! Get Cafe!)"
for file in `ls frames/$name/ |head -n -147`; do
	time python 2_deep_dreamv2.py frames/$name/$file processed_frames/$name/$file 2 8 0;
	# time python 2_deep_dreamv2.py test_done1.jpg test_done2.jpg 8 10 0
	# time python DreamOn.py frames/$name/$file processed_frames/$name/$file 6 4 10;
done

echo "Stitching the tripped out jpg frames back to a mp4 with sound of the original"
./3_frames2movie.sh ffmpeg processed_frames/$name/ $download_folder$filename $extension


# 249          webm       audio only DASH audio   65k , opus @ 50k, 1.57MiB
# 250          webm       audio only DASH audio   85k , opus @ 70k, 2.06MiB
# 140          m4a        aud1io only DASH audio  130k , m4a_dash container, mp4a.40.2@128k, 3.90MiB
# 171          webm       audio only DASH audio  141k , vorbis@128k, 3.60MiB
# 251          webm       audio only DASH audio  159k , opus @160k, 4.01MiB
# 278          webm       256x144    144p   96k , webm container, vp9, 25fps, video only, 2.57MiB
# 160          mp4        256x144    144p  112k , avc1.4d400c, 25fps, video only, 2.25MiB
# 242          webm       426x240    240p  223k , vp9, 25fps, video only, 4.77MiB
# 133          mp4        426x240    240p  250k , avc1.4d4015, 25fps, video only, 4.36MiB
# 243          webm       640x360    360p  411k , vp9, 25fps, video only, 8.65MiB
# 134          mp4        640x360    360p  640k , avc1.4d401e, 25fps, video only, 10.43MiB
# 244          webm       854x480    480p  755k , vp9, 25fps, video only, 14.74MiB
# 1353          mp4        854x480    480p 1183k , avc1.4d401e, 25fps, video only, 19.63MiB
# 247          webm       1280x720   720p 1506k , vp9, 25fps, video only, 28.75MiB
# 136          mp4        1280x720   720p 2036k , avc1.4d401f, 25fps, video only, 36.36MiB
# 18           mp4        640x360    medium , avc1.42001E, mp4a.40.2@ 96k, 17.51MiB
# 43           webm       640x360    medium , vp8.0, vorbis@128k, 21.38MiB
# 22           mp4        1280x720   hd720 , avc1.64001F, mp4a.40.2@192k (best)
