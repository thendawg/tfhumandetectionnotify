<!DOCTYPE html>
<html>
<head>
<?php
header('Cache-control: no-cache');
?>
<script>
window.setInterval(function()
{
   document.getElementById('mlcam').src = "detectimgs/RD/driveway.jpg?random="+new Date().getTime();
   document.getElementById('mlcam2').src = "detectimgs/RD/frontyard.jpg?random="+new Date().getTime();
}, 475);
</script>
<style>
body {
    background-color: black;
}
</style>
</head>
<body>
<center>
	<img id="mlcam" src="detectimgs/RD/drivewaytmp.jpg" \>
<div style="line-height:40%;">
    <br>
</div>
	<img id="mlcam2" src="detectimgs/RD/frontyardtmp.jpg" \>
</center>
</body>
</html>
