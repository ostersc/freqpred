interface Props {
  message: string
}

export default function ErrorBanner({ message }: Props) {
  return (
    <div className="rounded bg-red-50 border border-red-200 px-4 py-3 text-red-800 text-sm">
      {message}
    </div>
  )
}
