import { useState, useEffect } from 'react'
import DragSpace from './drag'

export default function Carousel() {
    const [images, setImages] = useState([])
    useEffect(() => {
        fetch('https://jsonplaceholder.typicode.com/photos')
            .then(response => response.json())
            .then(data => {
                let images = []
                let counter = 0
                for (const image of data){
                    images.push(image);
                    counter += 1;
                    if (counter > 2){
                        break
                        }
                    }
                return setImages(images)
            }
        )
    }, [])
    return (
    <div className="slideshow-container">
        {images.map(
            (image) => 
            <DragSpace key={image.id} className="carousel-img" idSelf={image.id}>
                <img src={image.url} alt={image.id} id={image.id} className="carousel-img"/>
                <div className="text">{image.id}</div>
            </DragSpace>
        )}
    </div>
    )
}
