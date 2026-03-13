import {BarChart, Bar, XAxis, YAxis, Tooltip} from 'recharts'

export default function ColourCharts({data}) {
    return (
        <BarChart width={600} height={300} data={data}>
            <XAxis dataKey="colour" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="percentage" fill="#8884d8" />
        </BarChart>
    )}