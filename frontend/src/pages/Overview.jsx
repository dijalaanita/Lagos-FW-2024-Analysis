import { useEffect, useState } from "react"
import { getOverviewColours } from "../services/api"
import ColourChart from "../components/ColourChart"
import Insights from "../components/Insights"

export default function Overview(){
    const [data, setData] = useState([])
    useEffect(() => {

        getOverviewColours().then((response) => setData(response.data))

    }, [])

    return (
        <div>
            <h1>LAGOS FASHION WEEK COLOUR TRENDS</h1>
            <ColourChart data={data}/>
            <Insights data={data}/>
            </div>
    )
}