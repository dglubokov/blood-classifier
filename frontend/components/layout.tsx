import Head from 'next/head'
import styles from './layout.module.css'

export default function Layout({ children }) {
    return (
    <div className={styles.container}>
        <Head>
            <title>Blood AI</title>
            <link rel="icon" href="/favicon.png" />
        </Head>
        <header className={styles.header}>
            Blood AI 🩸
        </header>
        <main>{children}</main>
    </div>
    )
}