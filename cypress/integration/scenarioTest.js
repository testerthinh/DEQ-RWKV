/// <reference types="Cypress" /> 
import "@4tw/cypress-drag-drop";
import "cypress-real-events"

describe('Test Scenario', function()
{
    const stepTo = ($el, target) => {
    const step = $el[0].getAttribute('step') || 1
    const current = $el[0].value
    const diff = target - current
    const steps = Math.abs(diff * step)
    if (diff > 0) {
        $el[0].stepUp(steps) }
    else {
        $el[0].stepDown(steps)
    }}
 
    it('Scenario 1',function()
    {
        cy.viewport(1280, 720) 
        cy.visit('https://www.lambdatest.com/selenium-playground').wait(2000)
        cy.contains('Drag & Drop Sliders').click().wait(1000)
        cy.get('#slider3 > div > input')
        .then($el => stepTo($el, 95) )  
        .trigger('change')
        cy.get('#rangeSuccess').should('have.value','95')


    })


    it('Scenario 2',function()
    {

        cy.viewport(414, 846)               // the screen of Samsung Note 9 
     //   cy.intercept('**').as('requests')
        cy.visit('https://www.lambdatest.com/selenium-playground')
      //  cy.wait('@requests')
        cy.contains('Input Form Submit').click().wait(1000)
        Cypress.on('uncaught:exception', (err, runnable) => {
            // returning false here prevents Cypress from
            // failing the test
            cy.injectAxe()
            cy.checkA11y()
            return false
        })

        
        cy.get('#name').click().realType('Thinh')
        cy.get('#inputEmail4').click().realType('thinhdd6@gmail.com')
        cy.get('#inputPassword4').click().realType('1234')
        cy.get('#company').click().realType('company')
        cy.get('#websitename').click().realType('lambdatest.com')
        cy.get(':nth-child(3) > .pr-20 > .w-full').select('American Samoa')            

        cy.get('#inputCity').click().realType('Ho Chi Minh')
        cy.get('#inputAddress1').click().realType('Address 1, Viet Nam')
        cy.get('#inputAddress2').click().realType('Address 2, Viet Nam')
        cy.get('#inputState').click().realType('Ho Chi Minh')
        cy.get('#inputZip').click().realType('70000')
        cy.contains('Submit').click().lighthouse(
            {
            performance: 70,
            accessibility: 90,
           'best-practices': 90,
            }
        )
        cy.get('.success-msg').should('have.text','Thanks for contacting us, we will get back to you shortly.')

    })

    this.afterAll('close browser', () => {
        cy.window().then((win) => {
            win.close()
        })
    })

})