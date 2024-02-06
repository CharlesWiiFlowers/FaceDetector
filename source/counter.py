import flet as ft

def main(page: ft.Page):
    page.title = "Demo of a counter"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    number = ft.TextField(value="0", text_align=ft.TextAlign.CENTER, width=100)

    def minus_click(e):
        number.value = str(int(number.value) - 1)
        page.update()

    def plus_click(e):
        number.value = str(int(number.value) + 1)
        page.update()

    page.add(
        ft.Row(
            [
                ft.IconButton(ft.icons.REMOVE, on_click=minus_click),
                number,
                ft.IconButton(ft.icons.ADD, on_click=plus_click),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )
    )
    
ft.app(target=main, view=ft.AppView.FLET_APP)