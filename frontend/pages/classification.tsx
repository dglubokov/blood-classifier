import Link from 'next/link'
import Layout from '../components/layout'

export default function Classification() {
    return (
    <Layout>
        <h1>Classification</h1>
        <h3>
            <Link href="/">
                <a>← back</a>
            </Link>
        </h3>
        <div className="main-buttons">
            <form className="form">
                <p> Выберите файл с изображениями в формате .zip</p>
                <div className="upload">
                    <label htmlFor="myfile" className="label">Выберите файл</label>
                    <input type="file" className="upload-button" id="myfile" name="myfile" multiple></input>
                    <button type="submit" className="submit">Загрузить 🔄 </button>
                </div>
            </form>

            <form className="form" id="downloading">
                <a href="http://127.0.0.1:8082/download/" download="results.tsv">
                    <button type="button" id="download" className="button">Выгрузить результаты ✅</button>
                </a>
            </form>

            <form className="form" id="summary">
                <a download="summary.png" href="http://127.0.0.1:8082/download-summary/" title="summary">
                    <button type="button" id="show_infer" className="button">Показать сводку 📊</button>
                </a>
                
            </form>
        </div>

        
    </Layout>
    )
}
