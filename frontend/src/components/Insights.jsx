export default function Insights({ data }){
    if (!data || !data.brand_colours || data.brand_colours.length == 0) return null

    const top = data.brand_colours[0]

    return (
        <div style={{ marginTop: "30px" }}>

            <h3>Key Insights</h3>
            <p>
                The dominant colour in this collection is <strong>{top.colour}</strong>
                appearing in <strong>{
                (top.percentage || 0).toFixed(1)}%</strong> of the looks.
            </p>

        </div>
    )

}