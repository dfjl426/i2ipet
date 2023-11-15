$(document).ready(function (e) {
    updateLabelStyle();

    $('.resultBtnDiv').css('display', 'none');
    $(".deleteImgBtn").css('display', 'none');

    $('.uploadImgBtn').on('click', function () {
        var form_data = new FormData();
        var ins = document.getElementById('selectImage').files.length;

        if(ins == 0) {
            $('#msg').html('<span style="color:red">🐶사진을 선택하세요🐶</span>');
            $('#msg').css('display', 'block');
            return;
        }

        form_data.append("selectImage", document.getElementById('selectImage').files[0]);

        // 로딩 창 표시
        $(".loading").css('display', 'flex');

        $.ajax({
            url: 'img2img', // point to server-side URL
            dataType: 'json', // what to expect back from server
            cache: false,
            contentType: false,
            processData: false,
            data: form_data,
            type: 'post',
            success: function (response) { // display success response
                // 로딩 창 숨김
                $(".loading").hide();

                $('#msg').html('');
                $('#previewImg').css('display','none');
                $('.uploadBtnDiv').css('display', 'none');
                $('.resultBtnDiv').css('display', 'block');
                $('#resultImg').attr('src', response.img_str);
            },
            error: function (response) {
                console.log(response.message); // display error response
            }
        }); 
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