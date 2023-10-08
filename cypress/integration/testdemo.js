/// <reference types="Cypress" /> 
import "cypress-real-events/support";
import "@4tw/cypress-drag-drop";

describe('Lighthouse', () => {
    it('should run performance audits', () => {
      cy.visit('https://www.browserstack.com/');

      cy.go('back')           

    });
  });

