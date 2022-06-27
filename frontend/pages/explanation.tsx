import Link from 'next/link'
import Layout from '../components/layout'
import Carousel from '../components/carousel'
import DragSpace from '../components/drag'
import CarouselBackup from '../components/carousel_backup'

export default function Explanation() {
    return (
    <Layout>

        <h1>Feature Explanation</h1>
        <h3>
            <Link href="/">
              <a>← back</a>
            </Link>
        </h3>
 
        <p>Поместите изображение:</p>
        <div className="explanation">
            <DragSpace className="expl-image-space" idSelf="expl-image-space"></DragSpace>
            <div className="expl-options">
                <form className="form">
                    <label htmlFor="models">Выберите модель: </label>
                    <br />
                    <select name="models" id="models">
                        <option value="root">Root</option>
                        <option value="ley">Лейкоциты</option>
                        <option value="patho">Патогенные</option>
                    </select>
                    <br/>
                    <input className="submit" type="submit" value="Запустить"/>
                </form>
            </div>
        </div>

        <br />
        
        <div className="main-buttons">
        {/* <Carousel></Carousel> */}
        <CarouselBackup></CarouselBackup>
            <form className="form">
                <div className="upload">
                    <label htmlFor="add-image-explanation" className="label">Добавить изображение</label>
                    <input 
                        type="file"
                        className="add-image upload-button"
                        id="add-image-explanation"
                        name="add-image-explanation"
                        multiple
                    ></input>
                </div>
            </form>
        </div>

    </Layout>
    )
}
