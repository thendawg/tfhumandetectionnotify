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
	<img id="mlcam" src="detectimgs/RD/drivewaytmp.jpg" height="250" width="367"  \>
<div style="line-height:40%;">
    <br>
</div>
	<img id="mlcam2" src="detectimgs/RD/frontyardtmp.jpg" height="250" width="367"  \>
</center>
<script>
document.getElementById("mlcam").addEventListener("error", ErrorDr);
function ErrorDr() {
document.getElementById("mlcam").src = "detectimgs/RD/driveway.jpg";
}
document.getElementById("mlcam2").addEventListener("error", ErrorYd);
function ErrorYd() {
document.getElementById("mlcam2").src = "detectimgs/RD/frontyard.jpg";
}
</script>
</body>
</html>
