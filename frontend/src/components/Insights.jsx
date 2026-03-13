export default function Insights({ data }){
    if (!data.length) return null

    const top = data[0]

    return (
        <div style={{ marginTop: "30px" }}>

            <h3>Key Insights</h3>
            <p>
                The dominant colour in this collection is <strong>{top.colour}</strong>
                appearing in <strong>{top.percentage.toFixed(1)}%</strong> of the looks.
            </p>

        </div>
    )

}