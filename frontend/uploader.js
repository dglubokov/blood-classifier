const imgLoaderForm = document.getElementById('imgLoaderForm');
const inputFile = document.getElementById('inputFile');

const downloading = document.getElementById('downloading')

imgLoaderForm.addEventListener('submit', e => {
    e.preventDefault();
    if (inputFile.files.length > 0) {
        let file = inputFile.files[0];
        let reader  = new FileReader();
        
        reader.onload = function(e)  {
            let image = document.getElementById('sample');
            image.src = e.target.result;
        }
        reader.readAsDataURL(file);

        const formData = new FormData();
        const endpoint = 'http://127.0.0.1:8082/upload/';
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

// downloading.addEventListener('submit', e => {
//     fetch('http://127.0.0.1:8082/download/')
//     .then(resp => resp.blob())
//     .then(blob => {
//       const url = window.URL.createObjectURL(blob);
//       const a = document.createElement('a');
//       a.style.display = 'none';
//       a.href = url;
//       // the filename you want
//       a.download = 'todo-1.json';
//       document.body.appendChild(a);
//       a.click();
//       window.URL.revokeObjectURL(url);
//       alert('your file has downloaded!'); // or you know, something with better UX...
//     })
//     .catch(() => alert('oh no!'));
// })

// function download(url, filename) {
//     fetch(url)
//       .then(response => response.blob())
//       .then(blob => {
//         const link = document.createElement("a");
//         link.href = URL.createObjectURL(blob);
//         link.download = filename;
//         link.click();
//     })
//     .catch(console.error);
// }