var socket = io.connect('http://' + document.domain + ':' + location.port);

$(document).ready(function (e) {
    updateLabelStyle();

    $('.resultBtnDiv').css('display', 'none');
    $(".deleteImgBtn").css('display', 'none');

    $('.uploadImgBtn').on('click', function () {
        var selectImage = $('#selectImage')[0].files;

        if (selectImage.length > 0) {
            $(".loading").css('display', 'flex');
            
            // var formData = new FormData();
            // formData.append('selectImage', selectImage[0]);
            
            socket.emit('upload', {'selectImage': selectImage[0]});

        }else{
            $('#msg').html('<span>🐶사진을 선택하세요🐶</span>');
            $('#msg').css('display', 'block');
        }
    });

    

    $('.deleteImgBtn').on('click', function () {
        // 이미지 초기화
        $("#selectImageLabel").css("display", "block");
        $("#imgDiv").css("display", "none");
        $("#msg").html(''); // 메시지 초기화
    
        // 업로드한 파일 초기화 (input[type=file] 리셋)
        $("#selectImage").val('');
    
        // 프리뷰 이미지 초기화
        $("#previewImg").attr("src", '');
        $(".deleteImgBtn").css('display', 'none');
    });

    $('.downloadImgBtn').on('click', function(){
        if ($('#resultImg').attr('src') !== "") {
            var currentTime = new Date();
            var fileName = 'govr_' + currentTime.getFullYear() +
                ('0' + (currentTime.getMonth() + 1)).slice(-2) +
                ('0' + currentTime.getDate()).slice(-2) +
                ('0' + currentTime.getHours()).slice(-2) +
                ('0' + currentTime.getMinutes()).slice(-2) +
                ('0' + currentTime.getSeconds()).slice(-2) +
                currentTime.getMilliseconds();

            // 이미지 다운로드 링크 생성
            var downloadLink = $('<a>');
            downloadLink.attr('href', $('#resultImg').attr('src'));
            downloadLink.attr('download', fileName + '.png');

            // 다운로드 링크를 body에 추가하고 클릭
            $('body').append(downloadLink);
            downloadLink[0].click();
            $('body').remove(downloadLink);
        }else{
            alert('이미지가 로드되지 않았습니다.');
        }
    });

    socket.on('result', function (response) {
        $(".loading").hide();
        if (response.message != '') {
            $('#msg').css('display', 'none');
            $('#previewImg').css('display', 'none');
            $('.uploadBtnDiv').css('display', 'none');
            $('.resultBtnDiv').css('display', 'block');
            $('#resultImg').attr('src', response.img_str);
        } else if (response.error != '') {
            $('#msg').html('<span>' + response.error + '</span>');
            $('#msg').css('display', 'block');
        }
    });

    var isCheckingOrigin = false;

    $('.checkOriginImgBtn').on('mousedown touchstart', function () {
        isCheckingOrigin = true;
        // previewImg를 활성화하고 resultImg를 비활성화
        $('#previewImg').css('display', 'block');
        $('#resultImg').css('display', 'none');
    });

    $(document).on('mouseup touchend', function () {
        if (isCheckingOrigin) {
            isCheckingOrigin = false;
            // previewImg를 비활성화하고 resultImg를 활성화
            $('#previewImg').css('display', 'none');
            $('#resultImg').css('display', 'block');
        }
    });

    $(".shareURL").click(function () {
        // URL을 공유하는 로직을 작성
        alert("Sharing URL...");
    });

    $(".shareToX").click(function () {
        // 트위터 API 또는 URL Scheme을 사용하여 트위터에 공유하는 로직을 작성
        alert("Sharing to Twitter...");
    });

    $(".shareToIG").click(function () {
        // 인스타그램 API 또는 URL Scheme을 사용하여 인스타그램에 공유하는 로직을 작성
        alert("Sharing to Instagram...");
    });
    
    $('.reloadBtn').on('click', function(){
        location.reload();
    });
});



function updateLabelStyle() {
    var label = $('#selectImageLabel');
    var width = label.width();
    label.css({
        height: width + 'px',
        lineHeight: width + 'px'
    });

    $('.uploadBtnDiv, .resultBtnDiv').css('width', width);
}

function loadFile(input) {
    var file = input.files[0];	//선택된 파일 가져오기

    if(file != null){    
        $("#previewImg").attr("src", URL.createObjectURL(file));
        $("#imgDiv").css("display", "block");
        $("#selectImageLabel").css("display", "none");
        $("#msg").css("display", "none");
        $(".deleteImgBtn").css('display', 'block');
    }
};

$(window).on('resize', function() {
    updateLabelStyle();
});