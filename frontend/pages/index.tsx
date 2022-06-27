import Link from 'next/link'
import Layout from '../components/layout'


export default function HomePage() {
  return (
    <div className="content">
    <Layout>
     <div className="description">
      Автоматизированная система поддержки принятия решений для анализа клеточного состава крови и костного мозга
      при помощи глубоких нейронных сетей.
     </div>

      <div className="main-buttons">
          <h2>Выберите режим работы</h2>

          <h2 className="title">
            <Link href="/detection">
            <button className="button">Обнаружение клеток</button>
            </Link>
          </h2>

          <h2 className="title">
            <Link href="/classification">
            <button className="button">Классификация</button>
            </Link>
          </h2>

          <h2 className="title">
            <Link href="/explanation">
              <button className="button">Интерпретация</button>
            </Link>
        </h2>
      </div>
    </Layout>
    </div>
  )
}