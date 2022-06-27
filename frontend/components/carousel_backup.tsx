import { useState, useEffect } from 'react'
import { Splide, SplideSlide, SplideTrack } from '@splidejs/react-splide';
import '@splidejs/splide/css';


export const CarouselBackup = () => {
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
                    if (counter > 3){
                        break
                        }
                    }
                return setImages(images)
            }
        )
    }, [])
      
    function drag(ev) {
        ev.dataTransfer.setData("text", ev.target.id);
    }

    return (
    <Splide 
        hasTrack={ false } aria-label="..."
        options={{
            rewind      : true,
            speed       : 200,
            rewindSpeed : 1000,
            perPage     : 3,
            focus       : 'center',
            gap         : '3rem',
            autoHeight  : false,
            autoWidth   : false,
            trimSpace   : false,
            drag: false
        }}
    >
        <SplideTrack>
            {images.map((image) => 
                <SplideSlide 
                    key={image.id}
                ><img src={image.url} alt="to_carousel" id={image.id} className="carousel-img" draggable={true} onDragStart={drag}/>
                </SplideSlide>)}
        </SplideTrack>
    </Splide>
    )
}

export default CarouselBackup