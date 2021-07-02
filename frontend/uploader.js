const imgLoaderForm = document.getElementById('imgLoaderForm');
const inpImage = document.getElementById('inpImage');

imgLoaderForm.addEventListener('submit', e => {
    e.preventDefault();
    if (inpImage.files.length > 0) {
        let file = inpImage.files[0];
        let reader  = new FileReader();
        
        reader.onload = function(e)  {
            let image = document.getElementById('sample');
            image.src = e.target.result;
        }
        reader.readAsDataURL(file);

        const formData = new FormData();
        const endpoint = 'http://127.0.0.1:8082/uploadfile/';
        formData.append('file', file);
        fetch(endpoint, {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(messages => {
            let result = document.getElementById('results');
            result.innerHTML = 'class: ' + messages.class_name;
        });
    }
})
