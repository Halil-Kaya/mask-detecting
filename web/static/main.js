var video = document.querySelector("#videoElement");
const ImageBox = document.getElementById("image-box")
const imgResultElement = document.getElementById("resultImg") 

function readURL(input) {
    if (input.files && input.files[0]) {                      // if input is file,files has content
        var inputFileData = input.files[0];                     // shortcut
        var reader = new FileReader();                          // FileReader() : init
        reader.onload = function (e) {                           /* FileReader : set up ************** */
            $('.file-upload-placeholder').hide();                 // call for action element : hide
            $('.file-upload-image').attr('src', e.target.result); // image element : set src data.
            $('.file-upload-preview').show();                     // image element's container : show
            $('.image-title').html(inputFileData.name);           // set image's title
        };
        reader.readAsDataURL(inputFileData);     // reads target inputFileData, launch `.onload` actions
    } else { removeUpload(); }
}

function removeUpload() {
    var $clone = $('.file-upload-input').val('').clone(true); // create empty clone
    $('.file-upload-input').replaceWith($clone);              // reset input: replaced by empty clone
    $('.file-upload-placeholder').show();                     // show placeholder
    $('.file-upload-preview').hide();                         // hide preview
}

// Style when drag-over
$('.file-upload-placeholder').bind('dragover', function () {
    $('.file-upload-placeholder').addClass('image-dropping');
});
$('.file-upload-placeholder').bind('dragleave', function () {
    $('.file-upload-placeholder').removeClass('image-dropping');
});

let file = null;
const fileElement = document.getElementById('file')
const submitElement = document.getElementById('submit')

fileElement.onchange = function (event) {
    file = event.target.files[0];
}

function myFunction(stat, stat2) {
    document.getElementById("panel").style.display = stat;
    document.getElementById("videoElement").setAttribute("autoplay", stat2)
    
}

async function showImg() {
    const uploadedImage = document.getElementById('uploadedImage')
    const result = await axios({
        method: 'post',
        url: '/proc-image',
        data: {
            imageString : uploadedImage.src
        }
    });  
    const imageResult = result.data
    imgResultElement.src = "data:image/jpg;base64," + imageResult
    ImageBox.style.display = "block"
}


function closeAllWindows(){
    ImageBox.style.display = "none"
}