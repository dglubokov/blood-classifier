import Link from 'next/link'
import Layout from '../components/layout'
import DragSpace from '../components/drag'

export default function Detection() {
    return (
    <Layout>
        <h1>Object Detection</h1>
        <h3>
            <Link href="/">
                <a>← back</a>
            </Link>
        </h3>
        <DragSpace className="expl-image-space" idSelf="expl-image-space"></DragSpace>
        <form className="form">
                <p> Выберите Изображение</p>
                <div className="upload">
                    <label htmlFor="myfile" className="label">Выберите файл</label>
                    <input type="file" className="upload-button" id="myfile" name="myfile" multiple></input>
                    <button type="submit" className="submit">Загрузить 🔄 </button>
                </div>
            </form>
    </Layout>
    )
}
