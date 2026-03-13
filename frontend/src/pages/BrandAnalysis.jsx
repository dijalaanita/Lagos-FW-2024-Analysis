import { use, useEffect, useState } from "react"
import {getBrandColours} from "../services/api"
import ColourCharts from "../components/colourcharts"

export default function BrandAnalysis() {
    const [brand, setBrand] = useState([])

    useEffect(() => {
        getBrandColours("babayo").then(response => setDataStartEndIndexes(response.data))
    }, [])

    return (
        <div>
            <h1>Brand Colour Analysis</h1>
            <ColourCharts data={data} />
        </div>
    )
}